package com.ib.client;

import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import java.lang.reflect.Field;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

class ScannerSubscription_excludeConvertible_39_0_Test {

    @ParameterizedTest
    @ValueSource(strings = { "true", "false", "yes", "no", "", "someValue" })
    void testExcludeConvertible(String c) {
        ScannerSubscription subscription = new ScannerSubscription();
        try {
            Field excludeConvertibleField = ScannerSubscription.class.getDeclaredField("m_excludeConvertible");
            excludeConvertibleField.setAccessible(true);
            subscription.excludeConvertible(c);
            assertEquals(c, excludeConvertibleField.get(subscription));
        } catch (NoSuchFieldException | IllegalAccessException e) {
            fail("Error accessing private field: " + e.getMessage());
        }
    }

    @Test
    void testExcludeConvertibleNull() {
        ScannerSubscription subscription = new ScannerSubscription();
        try {
            Field excludeConvertibleField = ScannerSubscription.class.getDeclaredField("m_excludeConvertible");
            excludeConvertibleField.setAccessible(true);
            subscription.excludeConvertible(null);
            assertNull(excludeConvertibleField.get(subscription));
        } catch (NoSuchFieldException | IllegalAccessException e) {
            fail("Error accessing private field: " + e.getMessage());
        }
    }
}
