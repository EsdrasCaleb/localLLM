package com.hf.sfm.crypt;

import java.lang.reflect.Method;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class Base64_byteArrayToAltBase64_4_0_Test {

    @Test
    public void testByteArrayToAltBase64() throws Exception {
        // Test case 1: Empty byte array
        byte[] emptyArray = {};
        String expectedEmptyResult = "";
        assertEquals(expectedEmptyResult, Base64.byteArrayToAltBase64(emptyArray));
        // Test case 2: Single byte array
        // 'A' in ASCII
        byte[] singleByte = { 65 };
        String expectedSingleByteResult = "!";
        assertEquals(expectedSingleByteResult, Base64.byteArrayToAltBase64(singleByte));
        // Test case 3: Multiple bytes array
        // "ABC" in ASCII
        byte[] multipleBytes = { 65, 66, 67 };
        String expectedMultipleBytesResult = "!\"#";
        assertEquals(expectedMultipleBytesResult, Base64.byteArrayToAltBase64(multipleBytes));
        // Test case 4: Array with padding requirement
        // "AB" in ASCII
        byte[] paddingRequired = { 65, 66 };
        String expectedPaddingResult = "!\"";
        assertEquals(expectedPaddingResult, Base64.byteArrayToAltBase64(paddingRequired));
        // Test case 5: Array with special characters
        byte[] specialChars = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
        String expectedSpecialCharsResult = "!!\"#$%&'()*,-./:;<=>?@";
        assertEquals(expectedSpecialCharsResult, Base64.byteArrayToAltBase64(specialChars));
        // Test case 6: Array with negative values
        byte[] negativeValues = { -1, -2, -3, -4, -5, -6, -7, -8, -9, -10 };
        String expectedNegativeValuesResult = "//////??????????";
        assertEquals(expectedNegativeValuesResult, Base64.byteArrayToAltBase64(negativeValues));
        // Test case 7: Large array
        byte[] largeArray = new byte[100];
        for (int i = 0; i < 100; i++) {
            largeArray[i] = (byte) i;
        }
        String expectedLargeArrayResult = "!!\"#$%&'()*,-./:;<=>?@abcdefghijklmnopqrstuvwxyz0123456789+/!\"#$%&'()*,-./:;<=>?@abcdefghijklmnopqrstuvwxyz0123456789+/!\"#$%&'()*,-./:;<=>?@abcdefghijklmnopqrstuvwxyz0123456789+/!\"#$%&'()*,-./:;<=>?@abcdefghijklmnopqrstuvwxyz0123456789+/!\"#$%&'()*,-./:;<=>?@ab";
        assertEquals(expectedLargeArrayResult, Base64.byteArrayToAltBase64(largeArray));
    }

    @Test
    public void testByteArrayToAltBase64WithReflection() throws Exception {
        // Accessing the private method _$23170 using reflection
        Method method = Base64.class.getDeclaredMethod("_$23170", byte[].class, boolean.class);
        method.setAccessible(true);
        // Test case 1: Using reflection to call the private method directly
        // "ABC" in ASCII
        byte[] testArray = { 65, 66, 67 };
        String expectedResult = "!\"#";
        String result = (String) method.invoke(null, testArray, true);
        assertEquals(expectedResult, result);
    }
}
