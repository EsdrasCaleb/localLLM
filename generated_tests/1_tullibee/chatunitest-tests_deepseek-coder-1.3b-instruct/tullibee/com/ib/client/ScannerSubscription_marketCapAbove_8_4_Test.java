package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_marketCapAbove_8_4_Test {

    @Test
    public void testMarketCapAbove() {
        ScannerSubscription subscription = new ScannerSubscription();
        subscription.numberOfRows(10);
        subscription.marketCapAbove(2000000000.00);
        subscription.maturityDateAbove("2020-01-01");
        subscription.couponRateAbove(0.10);
        subscription.abovePrice(1000.00);
        subscription.belowPrice(900.00);
        subscription.aboveVolume(1000);
        subscription.averageOptionVolumeAbove(100);
        subscription.marketCapBelow(1500000000.00);
        subscription.moodyRatingAbove("A");
        subscription.spRatingAbove("A");
        subscription.maturityDateBelow("2022-01-01");
        subscription.couponRateBelow(0.15);
        subscription.excludeConvertible("No");
        subscription.scannerSettingPairs("Setting1,Setting2");
        subscription.stockTypeFilter("Equity");
        double actual = subscription.marketCapAbove();
        double expected = 2000000000.00;
        assertEquals(expected, actual);
    }
}
