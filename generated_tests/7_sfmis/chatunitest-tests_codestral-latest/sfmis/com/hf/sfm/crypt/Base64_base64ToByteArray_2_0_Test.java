package com.hf.sfm.crypt;

import java.lang.reflect.Field;
import java.lang.reflect.Method;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

class Base64_base64ToByteArray_2_0_Test {

    @InjectMocks
    private Base64 base64;

    @BeforeEach
    void setUp() {
        MockitoAnnotations.openMocks(this);
    }

    @Test
    void testPrivateFields() throws Exception {
        // Test the private fields using reflection
        Field field23169 = Base64.class.getDeclaredField("_$23169");
        field23169.setAccessible(true);
        byte[] field23169Value = (byte[]) field23169.get(null);
        assertNotNull(field23169Value);
        Field field23168 = Base64.class.getDeclaredField("_$23168");
        field23168.setAccessible(true);
        byte[] field23168Value = (byte[]) field23168.get(null);
    }
}
