package com.hf.sfm.crypt;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

@ExtendWith(MockitoExtension.class)
public class Base64_byteArrayToAltBase64_4_1_Test {

    @Mock
    private Base64 focal;

    @InjectMocks
    private Base64 base64;

    @Test
    public void testByteArrayToAltBase64_NullByteArray_ThrowsNullPointerException() {
        byte[] input = null;
        assertThrows(NullPointerException.class, () -> base64.byteArrayToAltBase64(input));
    }
}
