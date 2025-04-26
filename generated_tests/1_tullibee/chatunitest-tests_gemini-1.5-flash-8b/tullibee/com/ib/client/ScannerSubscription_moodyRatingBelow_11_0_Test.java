package com.ib.client;

import java.lang.reflect.Field;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

class ScannerSubscription_moodyRatingBelow_11_0_Test {

    @Test
    void testMoodyRatingBelow_empty() {
        ScannerSubscription subscription = new ScannerSubscription();
        String result = subscription.moodyRatingBelow();
        assertNull(result);
    }

    @Test
    void testMoodyRatingBelow_populated() throws NoSuchFieldException, IllegalAccessException {
        ScannerSubscription subscription = new ScannerSubscription();
        Field moodyRatingBelowField = ScannerSubscription.class.getDeclaredField("m_moodyRatingBelow");
        moodyRatingBelowField.setAccessible(true);
        moodyRatingBelowField.set(subscription, "Aaa");
        String result = subscription.moodyRatingBelow();
        assertEquals("Aaa", result);
    }
}
