package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_abovePrice_25_3_Test {

    @Test
    public void abovePriceTest() {
        ScannerSubscription subscription = new ScannerSubscription();
        double price = 100.0;
        subscription.abovePrice(price);
        assertEquals(price, subscription.abovePrice());
    }
}
