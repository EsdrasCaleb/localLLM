package com.densebrain.rif.util;

import org.junit.jupiter.api.function.Executable;
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

class ObjectUtility_encodeBytes_1_3_Test {

    @Test
    void encodeBytes_nullInput_throwsNullPointerException() {
        assertThrows(NullPointerException.class, () -> ObjectUtility.encodeBytes(null));
    }

    @Test
    void encodeBytes_nonNullInput_returnsBase64EncodedString() {
        byte[] bytes = { 1, 2, 3, 4, 5 };
        String expected = "VGhpcyBpcyBhIHZlcnkgbG9uZyBzdHJpbmcgdGhhdCBoZWFk";
        String actual = ObjectUtility.encodeBytes(bytes);
        assertEquals(expected, actual);
    }
}
