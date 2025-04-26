package com.ib.client;

import java.lang.reflect.Field;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_spRatingBelow_34_0_Test {

    @Test
    void testSpRatingBelow_NullInput() {
        ScannerSubscription subscription = new ScannerSubscription();
        subscription.spRatingBelow(null);
        assertEquals(null, getFieldValue(subscription, "m_spRatingBelow"));
    }

    @Test
    void testSpRatingBelow_ValidInput() {
        ScannerSubscription subscription = new ScannerSubscription();
        String rating = "BBB+";
        subscription.spRatingBelow(rating);
        assertEquals(rating, getFieldValue(subscription, "m_spRatingBelow"));
    }

    @Test
    void testSpRatingBelow_EmptyStringInput() {
        ScannerSubscription subscription = new ScannerSubscription();
        subscription.spRatingBelow("");
        assertEquals("", getFieldValue(subscription, "m_spRatingBelow"));
    }

    @Test
    void testSpRatingBelow_OverwriteInput() {
        ScannerSubscription subscription = new ScannerSubscription();
        subscription.spRatingBelow("BBB+");
        subscription.spRatingBelow("AA-");
        assertEquals("AA-", getFieldValue(subscription, "m_spRatingBelow"));
    }

    private Object getFieldValue(Object obj, String fieldName) {
        try {
            Field field = obj.getClass().getDeclaredField(fieldName);
            field.setAccessible(true);
            return field.get(obj);
        } catch (NoSuchFieldException | IllegalAccessException e) {
            throw new RuntimeException(e);
        }
    }
}
