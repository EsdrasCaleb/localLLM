package com.ib.client;

import java.lang.reflect.Field;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_maturityDateBelow_36_0_Test {

    @Test
    void testMaturityDateBelow_validDate() {
        ScannerSubscription subscription = new ScannerSubscription();
        subscription.maturityDateBelow("2024-12-31");
        assertEquals("2024-12-31", getFieldValue(subscription, "m_maturityDateBelow"));
    }

    @Test
    void testMaturityDateBelow_nullDate() {
        ScannerSubscription subscription = new ScannerSubscription();
        subscription.maturityDateBelow(null);
        assertNull(getFieldValue(subscription, "m_maturityDateBelow"));
    }

    @Test
    void testMaturityDateBelow_emptyDate() {
        ScannerSubscription subscription = new ScannerSubscription();
        subscription.maturityDateBelow("");
        assertEquals("", getFieldValue(subscription, "m_maturityDateBelow"));
    }

    @Test
    void testMaturityDateBelow_invalidDate() {
        ScannerSubscription subscription = new ScannerSubscription();
        subscription.maturityDateBelow("invalid date");
        assertEquals("invalid date", getFieldValue(subscription, "m_maturityDateBelow"));
    }

    private String getFieldValue(ScannerSubscription subscription, String fieldName) {
        try {
            Field field = ScannerSubscription.class.getDeclaredField(fieldName);
            field.setAccessible(true);
            return (String) field.get(subscription);
        } catch (NoSuchFieldException | IllegalAccessException e) {
            e.printStackTrace();
            fail("Failed to access field: " + fieldName);
            // Should not reach here due to fail()
            return null;
        }
    }
}
