package com.hf.sfm.crypt;

import java.lang.reflect.Method;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class Base64_byteArrayToBase64_6_0_Test {

    @Test
    public void testByteArrayToBase64() throws Exception {
        // Test case 1: Empty byte array
        byte[] input1 = {};
        String expected1 = "";
        assertEquals(expected1, Base64.byteArrayToBase64(input1));
        // Test case 2: Single byte
        // 'A'
        byte[] input2 = { 65 };
        String expected2 = "QQ==";
        assertEquals(expected2, Base64.byteArrayToBase64(input2));
        // Test case 3: Two bytes
        // "AB"
        byte[] input3 = { 65, 66 };
        String expected3 = "QUI=";
        assertEquals(expected3, Base64.byteArrayToBase64(input3));
        // Test case 4: Three bytes
        // "ABC"
        byte[] input4 = { 65, 66, 67 };
        String expected4 = "QUJD";
        assertEquals(expected4, Base64.byteArrayToBase64(input4));
        // Test case 5: Arbitrary bytes
        byte[] input5 = { 10, 20, 30, 40, 50 };
        String expected5 = "CgwsKDI=";
        assertEquals(expected5, Base64.byteArrayToBase64(input5));
        // Test case 6: All ASCII characters
        byte[] input6 = new byte[128];
        for (int i = 0; i < 128; i++) {
            input6[i] = (byte) i;
        }
        String expected6 = "AAECAwQFBgcICQoLDA0ODxAREhMUFRYXGBkaGxwdHh8gISIjJCUmJygpKissLS4vMDEyMzQ1Njc4OTo7PD0+P0BBQkNERUZHSElKS0xNTk9QUVJTVFVWV1hZWltcXV5fYGFiY2RlZmdoaWprbG1ub3BxcnN0dXZ3eHl6e3x9fn+AgYKDhIWGh4iJiouMjY6PkJGSk45OTk5OTk5OTk5OTk5OTk5OTk5OTk5OTk5OTk5OTk5OTk5OTk5OTk5OTk5OTk5OTk5OTk5OTk5OTk5OTk5OTk5OTk5OTk5OTk5OTk5OTk5OTk5OTk5OTk5OTk5OTk5OTk5OTk5OTk5OTk5OTk5OTk5OTk5OTk5OTk5OTk5OTk5OTk5OTk5OTk5OTk5OTk5OTk5OTk5OTk5OTk5OTk5OTk5OTk5OTk5OTk5OTk5OTk5OTk5OTk5OTk5OTk5OTk5OTk5OTk5OTk5OTk5OTk5OTk5OTk5OTk5AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA";
        assertEquals(expected6, Base64.byteArrayToBase64(input6));
        // Test case 7: Bytes with padding
        byte[] input7 = { 1, 2, 3, 4 };
        String expected7 = "AQIDBA==";
        assertEquals(expected7, Base64.byteArrayToBase64(input7));
        // Test case 8: Bytes with different padding
        byte[] input8 = { 1, 2 };
        String expected8 = "AQI=";
        assertEquals(expected8, Base64.byteArrayToBase64(input8));
        // Test case 9: Negative bytes
        byte[] input9 = { -1, -2, -3, -4 };
        String expected9 = "/v8AAAAAAAA=";
        assertEquals(expected9, Base64.byteArrayToBase64(input9));
    }
}
