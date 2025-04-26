package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

class ScannerSubscription_abovePrice_4_0_Test {

    @Test
    void testAbovePrice() {
        ScannerSubscription subscription = new ScannerSubscription();
        double expectedAbovePrice = 100.00;
        subscription.abovePrice(expectedAbovePrice);
        double actualAbovePrice = subscription.abovePrice();
        assertEquals(expectedAbovePrice, actualAbovePrice);
    }
}
