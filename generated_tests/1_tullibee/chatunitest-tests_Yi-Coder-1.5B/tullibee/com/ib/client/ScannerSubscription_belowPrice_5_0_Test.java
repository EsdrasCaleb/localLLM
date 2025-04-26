package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

class ScannerSubscription_belowPrice_5_0_Test {

    @Test
    void belowPrice() {
        ScannerSubscription s = new ScannerSubscription();
        s.belowPrice(1.99);
        assertEquals(1.99, s.belowPrice(), 0.01);
    }
}
