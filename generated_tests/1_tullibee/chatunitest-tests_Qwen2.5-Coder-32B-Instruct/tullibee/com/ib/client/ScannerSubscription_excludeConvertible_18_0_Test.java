package com.ib.client;

import java.lang.reflect.Field;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_excludeConvertible_18_0_Test {

    private ScannerSubscription scannerSubscription;

    @BeforeEach
    public void setUp() {
        scannerSubscription = new ScannerSubscription();
    }

    @Test
    public void testExcludeConvertible_DefaultValue() {
        // Given: The default value of m_excludeConvertible is null
        // When: excludeConvertible() is called
        String result = scannerSubscription.excludeConvertible();
        // Then: It should return null
        assertNull(result);
    }

    @Test
    public void testExcludeConvertible_SetValue() throws NoSuchFieldException, IllegalAccessException {
        // Given: Set a value to m_excludeConvertible using reflection
        Field excludeConvertibleField = ScannerSubscription.class.getDeclaredField("m_excludeConvertible");
        excludeConvertibleField.setAccessible(true);
        excludeConvertibleField.set(scannerSubscription, "ExcludeConvertibleValue");
        // When: excludeConvertible() is called
        String result = scannerSubscription.excludeConvertible();
        // Then: It should return the set value
        assertEquals("ExcludeConvertibleValue", result);
    }
}
