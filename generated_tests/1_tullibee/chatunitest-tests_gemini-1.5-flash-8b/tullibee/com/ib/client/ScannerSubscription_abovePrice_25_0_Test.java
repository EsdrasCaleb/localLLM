package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

class ScannerSubscription_abovePrice_25_0_Test {

    @Test
    void abovePrice_validInput() {
        ScannerSubscription subscription = new ScannerSubscription();
        double price = 100.50;
        subscription.abovePrice(price);
        assertEquals(price, subscription.abovePrice());
    }

    @Test
    void abovePrice_zeroInput() {
        ScannerSubscription subscription = new ScannerSubscription();
        double price = 0;
        subscription.abovePrice(price);
        assertEquals(price, subscription.abovePrice());
    }

    @Test
    void abovePrice_negativeInput() {
        ScannerSubscription subscription = new ScannerSubscription();
        double price = -10.25;
        subscription.abovePrice(price);
        assertEquals(price, subscription.abovePrice());
    }

    @Test
    void abovePrice_maxDoubleInput() {
        ScannerSubscription subscription = new ScannerSubscription();
        double price = Double.MAX_VALUE;
        subscription.abovePrice(price);
        assertEquals(price, subscription.abovePrice());
    }

    @Test
    void abovePrice_minDoubleInput() {
        ScannerSubscription subscription = new ScannerSubscription();
        double price = Double.MIN_VALUE;
        subscription.abovePrice(price);
        assertEquals(price, subscription.abovePrice());
    }
}
