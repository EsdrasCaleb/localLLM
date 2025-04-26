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
    void altBase64ToByteArray_validInput_returnsCorrectByteArray() {
        String validBase64String = "SGVsbG8gV29ybGQh";
        byte[] expectedByteArray = { 72, 101, 108, 108, 111, 32, 87, 111, 114, 108, 100, 33 };
        byte[] actualByteArray = Base64.altBase64ToByteArray(validBase64String);
        assertArrayEquals(expectedByteArray, actualByteArray);
    }

    @Test
    void altBase64ToByteArray_emptyInput_returnsEmptyByteArray() {
        String emptyString = "";
        byte[] expectedByteArray = {};
        byte[] actualByteArray = Base64.altBase64ToByteArray(emptyString);
        assertArrayEquals(expectedByteArray, actualByteArray);
    }

    @Test
    void altBase64ToByteArray_nullInput_returnsNull() {
        String nullString = null;
        byte[] actualByteArray = Base64.altBase64ToByteArray(nullString);
        assertNull(actualByteArray);
    }

    @Test
    void altBase64ToByteArray_invalidInput_returnsEmptyByteArray() {
        // Invalid character
        String invalidBase64String = "SGVsbG8gV29ybGQhX";
        byte[] expectedByteArray = {};
        byte[] actualByteArray = Base64.altBase64ToByteArray(invalidBase64String);
        assertArrayEquals(expectedByteArray, actualByteArray);
    }

    @Test
    void altBase64ToByteArray_shortInput_returnsCorrectByteArray() {
        // Short string
        String shortBase64String = "SGVsbG8";
        byte[] expectedByteArray = { 72, 101, 108, 108, 111, 108 };
        byte[] actualByteArray = Base64.altBase64ToByteArray(shortBase64String);
        assertArrayEquals(expectedByteArray, actualByteArray);
    }
}
