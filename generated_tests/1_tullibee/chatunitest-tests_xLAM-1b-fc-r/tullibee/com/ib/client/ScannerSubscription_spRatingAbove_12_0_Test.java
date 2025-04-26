package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_spRatingAbove_12_0_Test {

    @Test
    public void spRatingAboveTest() {
        ScannerSubscription scanner = new ScannerSubscription();
        scanner.spRatingAbove("Test");
        assertEquals("Test", scanner.spRatingAbove());
    }
}
