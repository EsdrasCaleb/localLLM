package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_abovePrice_25_0_Test {

    @Test
    void abovePrice() {
        ScannerSubscription sut = new ScannerSubscription();
        sut.abovePrice(10.0);
        assertEquals(10.0, sut.abovePrice(), "Expected value is 10.0");
    }
}
