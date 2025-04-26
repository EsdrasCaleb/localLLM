package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_abovePrice_25_0_Test {

    @Test
    public void testAbovePrice() {
        ScannerSubscription scannerSubscription = new ScannerSubscription();
        double testPrice = 100.0;
        scannerSubscription.abovePrice(testPrice);
        assertEquals(testPrice, scannerSubscription.abovePrice());
    }
}
