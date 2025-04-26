package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_belowPrice_5_0_Test {

    @Test
    public void testBelowPrice_ReturnsInitialValue() {
        ScannerSubscription subscription = new ScannerSubscription();
        double result = subscription.belowPrice();
        assertEquals(Double.MAX_VALUE, result);
    }

    @Test
    public void testBelowPrice_SetAndGet() {
        ScannerSubscription subscription = new ScannerSubscription();
        subscription.belowPrice(100.0);
        double result = subscription.belowPrice();
        assertEquals(100.0, result);
    }

    @Test
    public void testBelowPrice_ThrowsException() {
        ScannerSubscription subscription = new ScannerSubscription();
        assertThrows(NullPointerException.class, () -> subscription.belowPrice());
    }
}
