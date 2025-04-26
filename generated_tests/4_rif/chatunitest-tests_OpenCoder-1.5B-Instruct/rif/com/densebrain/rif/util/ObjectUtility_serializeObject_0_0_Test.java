// Test method
package com.densebrain.rif.util;

import java.io.Serializable;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import org.apache.axis2.util.Base64;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

public class ObjectUtility_serializeObject_0_0_Test {

    @Test
    public void testSerializeObject() throws IOException, ClassNotFoundException {
        SomeClass someClass = new SomeClass("TestName");
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        ObjectOutputStream oos = new ObjectOutputStream(baos);
        oos.writeObject(someClass);
        byte[] serializedObj = baos.toByteArray();
        ByteArrayInputStream bais = new ByteArrayInputStream(serializedObj);
        ObjectInputStream ois = new ObjectInputStream(bais);
        SomeClass deserializedObj = (SomeClass) ois.readObject();
        assertEquals("TestName", deserializedObj.getName());
    }
}

class SomeClass implements Serializable {

    private String name;

    public SomeClass(String name) {
        this.name = name;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }
}
