package com.hf.sfm.crypt;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

class Base64_byteArrayToBase64_6_0_Test {

    @Test
    void testByteArrayToBase64() {
        byte[] bb = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };
        String expected = "MTIzNDU2Nzg5Y2hhbmdlbmNl";
        String actual = Base64.byteArrayToBase64(bb);
        Assertions.assertEquals(expected, actual);
    }
}
