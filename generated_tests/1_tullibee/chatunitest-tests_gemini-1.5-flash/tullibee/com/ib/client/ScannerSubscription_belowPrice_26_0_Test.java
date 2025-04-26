package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_belowPrice_26_0_Test {

    @Test
    void testBelowPrice() {
        ScannerSubscription subscription = new ScannerSubscription();
        // Test with a valid price
        double validPrice = 100.50;
        subscription.belowPrice(validPrice);
        assertEquals(validPrice, subscription.belowPrice());
        // Test with a price of zero
        double zeroPrice = 0.0;
        subscription.belowPrice(zeroPrice);
        assertEquals(zeroPrice, subscription.belowPrice());
        // Test with a negative price (edge case)
        double negativePrice = -50.0;
        subscription.belowPrice(negativePrice);
        assertEquals(negativePrice, subscription.belowPrice());
        // Test with Double.MAX_VALUE (edge case)
        subscription.belowPrice(Double.MAX_VALUE);
        assertEquals(Double.MAX_VALUE, subscription.belowPrice());
        // Test with Double.MIN_VALUE (edge case)
        subscription.belowPrice(Double.MIN_VALUE);
        assertEquals(Double.MIN_VALUE, subscription.belowPrice());
    }
}
