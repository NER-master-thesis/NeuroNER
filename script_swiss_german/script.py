folder = self.parameters["wikidataNER_folder"]
method = self.parameters["method"]
language = self.parameters["language"]
keep_table = self.parameters["keep_table"]
id = self.parameters["filter"]
if "subpart" in self.parameters:
    subpart = self.parameters["subpart"]
    input_file = os.path.join(folder, language, method,
                              "wikipedia_dataset_{0}{1}_{2}/combined_{3}_{1}_{2}.txt".format(
                                  "with_tables_" if keep_table else "", str(float(id)), subpart, method))
    output_file = os.path.join(folder, language, method,
                               "wikipedia_dataset_{0}{1}_{2}/combined_processed_{3}_{1}_{2}.txt".format(
                                   "with_tables_" if keep_table else "", str(float(id)), subpart, method))

else:

    input_file = os.path.join(folder, language, method,
                              "wikipedia_dataset_{0}{1}/combined_{2}_{1}.txt".format(
                                  "with_tables_" if keep_table else "", str(float(id)), method))
    output_file = os.path.join(folder, language, method,
                               "wikipedia_dataset_{0}{1}/combined_processed_{2}_{1}.txt".format(
                                   "with_tables_" if keep_table else "", str(float(id)), method))

als_to_de = get_mapping_DE_ALS()
output = []
with codecs.open(input_file, "r", "utf-8") as file:
    for line in file:
        line = line.strip()
        if line:
            line = line.split()
            token = line[0]
            pos = line[1]
            tag = line[2]
            link = " ".join(line[3:]) if len(line) > 3 else ""
            capi = is_capitalized(token)
            if token.lower() in als_to_de:
                new_token = als_to_de[token.lower()]
                if capi:
                    new_token = new_token.capitalize()
            else:
                new_token = convert(token)
            if len(new_token.split()) > 1:
                if tag == 'O' or "I-" in tag:
                    for new_t in new_token.split():
                        output.append(" ".join([new_t, pos, tag, link]))

                else:
                    bio, class_ = tag.split("-")
                    for new_t in new_token.split():
                        output.append(" ".join([new_t, pos, tag, link]))
                        tag = "I-" + class_
            else:
                output.append(" ".join([new_token, pos, tag, link]))
        else:
            output.append(" ")
with codecs.open(output_file, "w", "utf-8") as file:
    file.write("\n".join(output))