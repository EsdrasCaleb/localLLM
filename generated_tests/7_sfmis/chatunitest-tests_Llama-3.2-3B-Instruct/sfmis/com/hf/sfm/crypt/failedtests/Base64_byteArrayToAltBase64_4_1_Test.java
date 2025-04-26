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
    public void testByteArrayToAltBase64_EmptyByteArray_EmptyString() {
        byte[] input = new byte[0];
        String expected = "";
        when(focal.byteArrayToAltBase64(input)).thenReturn(expected);
        String actual = base64.byteArrayToAltBase64(input);
        assertEquals(expected, actual);
    }

    @Test
    public void testByteArrayToAltBase64_NullByteArray_ThrowsNullPointerException() {
        byte[] input = null;
        assertThrows(NullPointerException.class, () -> base64.byteArrayToAltBase64(input));
    }

    @Test
    public void testByteArrayToAltBase64_SingleByteByteArray_SingleBase64Character() {
        byte[] input = { 0x12 };
        String expected = "w";
        when(focal.byteArrayToAltBase64(input)).thenReturn(expected);
        String actual = base64.byteArrayToAltBase64(input);
        assertEquals(expected, actual);
    }

    @Test
    public void testByteArrayToAltBase64_MultipleByteByteArray_AltBase64EncodedString() {
        byte[] input = { 0x12, 0x34, 0x56, 0x78 };
        String expected = "wSg==";
        when(focal.byteArrayToAltBase64(input)).thenReturn(expected);
        String actual = base64.byteArrayToAltBase64(input);
        assertEquals(expected, actual);
    }
}
