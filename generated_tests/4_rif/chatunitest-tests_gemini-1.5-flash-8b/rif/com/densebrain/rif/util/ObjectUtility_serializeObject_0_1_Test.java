package com.densebrain.rif.util;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.ObjectOutputStream;
import java.io.ObjectInputStream;
import java.io.ByteArrayInputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.Arrays;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.apache.axis2.util.Base64;

class ObjectUtility_serializeObject_0_1_Test {

    @Test
    void testSerializeObject_nullObject() throws IOException {
        byte[] result = ObjectUtility.serializeObject(null);
        Assertions.assertArrayEquals(new byte[0], result);
    }

    @Test
    void testSerializeObject_emptyObject() throws IOException {
        List<String> emptyList = new ArrayList<>();
        byte[] result = ObjectUtility.serializeObject(emptyList);
        Assertions.assertNotNull(result);
        Assertions.assertNotEquals(0, result.length);
    }

    @Test
    void testSerializeObject_stringObject() throws IOException {
        String str = "Hello, world!";
        byte[] result = ObjectUtility.serializeObject(str);
        Assertions.assertNotNull(result);
        Assertions.assertNotEquals(0, result.length);
    }

    @Test
    void testSerializeObject_integerObject() throws IOException {
        Integer num = 123;
        byte[] result = ObjectUtility.serializeObject(num);
        Assertions.assertNotNull(result);
        Assertions.assertNotEquals(0, result.length);
    }

    @Test
    void testSerializeObject_complexObject() throws IOException {
        List<String> list = new ArrayList<>(Arrays.asList("one", "two", "three"));
        list.add(null);
        byte[] result = ObjectUtility.serializeObject(list);
        Assertions.assertNotNull(result);
        Assertions.assertNotEquals(0, result.length);
    }
}
