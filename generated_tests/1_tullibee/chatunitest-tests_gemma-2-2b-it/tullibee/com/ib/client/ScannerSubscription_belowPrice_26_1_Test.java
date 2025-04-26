package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_belowPrice_26_1_Test {

    @Test
    void belowPrice() {
        ScannerSubscription scannerSubscription = new ScannerSubscription();
        scannerSubscription.belowPrice(10.0);
        assertEquals(10.0, scannerSubscription.belowPrice());
    }
}
