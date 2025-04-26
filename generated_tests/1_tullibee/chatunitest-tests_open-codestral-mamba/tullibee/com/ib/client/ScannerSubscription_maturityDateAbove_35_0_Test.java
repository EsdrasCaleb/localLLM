package com.ib.client;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

@ExtendWith(MockitoExtension.class)
class ScannerSubscription_maturityDateAbove_35_0_Test {

    @Test
    void testMaturityDateAbove() {
        ScannerSubscription subscription = Mockito.spy(new ScannerSubscription());
        String maturityDate = "2022-12-31";
        subscription.maturityDateAbove(maturityDate);
        assertEquals(maturityDate, subscription.maturityDateAbove());
    }
}
