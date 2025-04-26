package com.densebrain.rif.util;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import org.apache.axis2.util.Base64;

public class ObjectUtility_encodeBytes_1_4_Test {

    private ObjectUtility objectUtility = Mockito.mock(ObjectUtility.class);

    @Test
    public void testEncodeBytes() {
        byte[] bytes = { 72, 101, 108, 108, 111 };
        String expected = "SGVsbG8gV29ybGQ=";
        String actual;
        try {
            actual = ObjectUtility.encodeBytes(bytes);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
        Mockito.verify(objectUtility).encodeBytes(bytes);
        assert actual.equals(expected);
    }
}
