package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_excludeConvertible_39_0_Test {

    private ScannerSubscription scannerSubscription;

    @BeforeEach
    public void setUp() {
        scannerSubscription = new ScannerSubscription();
    }

    @Test
    public void testExcludeConvertible_WithValidInput() {
        String expectedValue = "exclude_this_convertible";
        scannerSubscription.excludeConvertible(expectedValue);
        // Use reflection to access the private field m_excludeConvertible
        String actualValue = getPrivateField(scannerSubscription, "m_excludeConvertible");
        assertEquals(expectedValue, actualValue);
    }

    @Test
    public void testExcludeConvertible_WithNullInput() {
        scannerSubscription.excludeConvertible(null);
        // Use reflection to access the private field m_excludeConvertible
        String actualValue = getPrivateField(scannerSubscription, "m_excludeConvertible");
        assertEquals(null, actualValue);
    }

    private String getPrivateField(ScannerSubscription scannerSubscription, String fieldName) {
        try {
            java.lang.reflect.Field field = ScannerSubscription.class.getDeclaredField(fieldName);
            field.setAccessible(true);
            return (String) field.get(scannerSubscription);
        } catch (NoSuchFieldException | IllegalAccessException e) {
            throw new RuntimeException(e);
        }
    }
}
