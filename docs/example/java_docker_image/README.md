## 🚀 Java 8 Docker Environment for Legacy Java Applications

This repository provides a Dockerized environment to run Java 1.8 applications that may not be compatible with newer Java versions (like JDK 11+). It’s especially useful for tools or libraries that rely on packages like `javax.xml.bind`, which were removed in later JDK versions.


---

### 🐳 Build the Docker Image

```bash
docker build -t java8-env .
```

---

### ▶️ Run the Application

```bash
docker run --rm -v D:/Projects/Mini/AI/BPS/DeclarativeProcessSimulation:/app -w /app java8-xvfb sh -c Xvfb :99 -screen 0 1024x768x16 & java -cp "GenerativeLSTM/external_tools/splitminer3/bpmtk.jar:GenerativeLSTM/external_tools/splitminer3/lib/*" au.edu.unimelb.services.ServiceProvider SMD 0.5 0.7 false false false GenerativeLSTM/input_files/spmd/RunningExample.xes GenerativeLSTM/input_files/spmd/RunningExample
```

> 🔁 On **Windows CMD**, replace `:` in the classpath with `;`.  
> 🐧 Inside the container (Linux), use `:` as the separator.

---

### ✅ Why Docker?

- Avoids compatibility issues with newer Java versions.
- Easily replicable across systems.
- Keeps your system clean from multiple JDK installs.
- Allows inclusion of missing modules like JAXB in a self-contained way.

---

### 🧩 Notes

- Be sure all required JAR files (including `jaxb-api` and `jaxb-runtime`) are placed in `GenerativeLSTM/external_tools/splitminer3/lib/`.
- You can download these from Maven Central or manually include them from a Java 8 environment.
- The app assumes Java class `au.edu.unimelb.services.ServiceProvider` is your main entry point.

---

Need a version with `docker-compose.yml` as well? I can add that too.