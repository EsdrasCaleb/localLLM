package com.ib.client;

import java.lang.reflect.Field;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

class ScannerSubscription_belowPrice_5_0_Test {

    @Test
    void testBelowPrice_validValue() {
        ScannerSubscription subscription = new ScannerSubscription();
        subscription.belowPrice(10.5);
        assertEquals(10.5, subscription.belowPrice());
    }

    @Test
    void testBelowPrice_defaultValue() {
        ScannerSubscription subscription = new ScannerSubscription();
        assertEquals(Double.MAX_VALUE, subscription.belowPrice());
    }

    @Test
    void testBelowPrice_zero() {
        ScannerSubscription subscription = new ScannerSubscription();
        subscription.belowPrice(0.0);
        assertEquals(0.0, subscription.belowPrice());
    }

    @Test
    void testBelowPrice_negativeValue() {
        ScannerSubscription subscription = new ScannerSubscription();
        subscription.belowPrice(-5.2);
        assertEquals(-5.2, subscription.belowPrice());
    }
}
