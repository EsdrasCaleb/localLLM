package com.hf.sfm.crypt;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class Base64_altBase64ToByteArray_0_3_Test {

    @InjectMocks
    private Base64 base64;

    @BeforeEach
    public void setUp() {
        MockitoAnnotations.openMocks(this);
    }

    @Test
    public void testAltBase64ToByteArrayWithInvalidInput() {
        String invalidEncodedString = "InvalidBase64String";
        assertThrows(IllegalArgumentException.class, () -> {
            Base64.altBase64ToByteArray(invalidEncodedString);
        });
    }

    @Test
    public void testAltBase64ToByteArrayWithEmptyString() {
        String emptyString = "";
        byte[] result = Base64.altBase64ToByteArray(emptyString);
        assertArrayEquals(new byte[0], result);
    }

    @Test
    public void testAltBase64ToByteArrayWithNullInput() {
        assertThrows(NullPointerException.class, () -> {
            Base64.altBase64ToByteArray(null);
        });
    }
}
