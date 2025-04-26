package com.hf.sfm.crypt;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class Base64_byteArrayToAltBase64_4_0_Test {

    @Test
    void byteArrayToAltBase64_shouldReturnAltBase64String() {
        byte[] bytes = { 1, 2, 3, 4, 5, 6, 7, 8 };
        String expected = "SGVsbG8sIHdvcmxkIQ==";
        String actual = Base64.byteArrayToAltBase64(bytes);
        assertEquals(expected, actual);
    }
}
