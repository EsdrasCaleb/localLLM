package com.hf.sfm.crypt;

import java.util.Arrays;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

class Base64_altBase64ToByteArray_0_1_Test {

    @Test
    void altBase64ToByteArray_emptyInput_returnsEmptyByteArray() {
        String emptyString = "";
        byte[] expectedByteArray = {};
        byte[] actualByteArray = Base64.altBase64ToByteArray(emptyString);
        assertArrayEquals(expectedByteArray, actualByteArray);
    }
}
