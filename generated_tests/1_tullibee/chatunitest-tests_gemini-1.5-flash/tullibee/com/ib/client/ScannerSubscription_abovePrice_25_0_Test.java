package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_abovePrice_25_0_Test {

    @Test
    void testAbovePrice_ValidPrice() {
        ScannerSubscription subscription = new ScannerSubscription();
        double price = 100.50;
        subscription.abovePrice(price);
        assertEquals(price, subscription.abovePrice());
    }

    @Test
    void testAbovePrice_MaxPrice() {
        ScannerSubscription subscription = new ScannerSubscription();
        subscription.abovePrice(Double.MAX_VALUE);
        assertEquals(Double.MAX_VALUE, subscription.abovePrice());
    }

    @Test
    void testAbovePrice_ZeroPrice() {
        ScannerSubscription subscription = new ScannerSubscription();
        subscription.abovePrice(0);
        assertEquals(0, subscription.abovePrice());
    }

    @Test
    void testAbovePrice_NegativePrice() {
        ScannerSubscription subscription = new ScannerSubscription();
        subscription.abovePrice(-100);
        assertEquals(-100, subscription.abovePrice());
    }

    @Test
    void testAbovePrice_MinValue() {
        ScannerSubscription subscription = new ScannerSubscription();
        subscription.abovePrice(Double.MIN_VALUE);
        assertEquals(Double.MIN_VALUE, subscription.abovePrice());
    }
}
