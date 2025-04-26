package com.ib.client;

import java.lang.reflect.Field;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

class ScannerSubscription_moodyRatingAbove_10_0_Test {

    @Test
    void testMoodyRatingAbove_validInput() {
        ScannerSubscription subscription = new ScannerSubscription();
        subscription.moodyRatingAbove("Aaa");
        assertEquals("Aaa", subscription.moodyRatingAbove());
    }

    @Test
    void testMoodyRatingAbove_nullInput() {
        ScannerSubscription subscription = new ScannerSubscription();
        subscription.moodyRatingAbove(null);
        assertNull(subscription.moodyRatingAbove());
    }

    @Test
    void testMoodyRatingAbove_emptyInput() {
        ScannerSubscription subscription = new ScannerSubscription();
        subscription.moodyRatingAbove("");
        assertEquals("", subscription.moodyRatingAbove());
    }

    @Test
    void testMoodyRatingAbove_initialDefault() {
        ScannerSubscription subscription = new ScannerSubscription();
        assertNull(subscription.moodyRatingAbove());
    }
}
