package com.hf.sfm.crypt;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class Base64_altBase64ToByteArray_0_3_Test {

    @Test
    public void testAltBase64ToByteArray_NullString() {
        String input = null;
        try {
            Base64.altBase64ToByteArray(input);
            fail("Expected NullPointerException");
        } catch (NullPointerException e) {
            // expected
        }
    }
}
