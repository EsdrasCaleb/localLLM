package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

class ScannerSubscription_belowPrice_26_0_Test {

    private ScannerSubscription scannerSubscription;

    @BeforeEach
    public void setup() {
        scannerSubscription = new ScannerSubscription();
    }

    @Test
    void belowPriceTest() {
        double newPrice = 100.0;
        double expectedPrice = scannerSubscription.belowPrice();
        scannerSubscription.belowPrice(newPrice);
        assertEquals(newPrice, scannerSubscription.belowPrice());
    }
}
