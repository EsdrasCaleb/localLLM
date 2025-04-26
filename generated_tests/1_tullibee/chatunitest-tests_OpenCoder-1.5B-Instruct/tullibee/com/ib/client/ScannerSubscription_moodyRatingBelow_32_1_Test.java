package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

// Focal class
public class ScannerSubscription_moodyRatingBelow_32_1_Test {

    // Focal class
    public class ScannerSubscription {

        public final static int NO_ROW_NUMBER_SPECIFIED = -1;

        private int m_numberOfRows = NO_ROW_NUMBER_SPECIFIED;

        private String m_instrument;

        private String m_locationCode;

        private String m_scanCode;

        private double m_abovePrice = Double.MAX_VALUE;

        private double m_belowPrice = Double.MAX_VALUE;

        private int m_aboveVolume = Integer.MAX_VALUE;

        private int m_averageOptionVolumeAbove = Integer.MAX_VALUE;

        private double m_marketCapAbove = Double.MAX_VALUE;

        private double m_marketCapBelow = Double.MAX_VALUE;

        private String m_moodyRatingAbove;

        private String m_moodyRatingBelow;

        private String m_spRatingAbove;

        private String m_spRatingBelow;

        private String m_maturityDateAbove;

        private String m_maturityDateBelow;

        private double m_couponRateAbove = Double.MAX_VALUE;

        private double m_couponRateBelow = Double.MAX_VALUE;

        private String m_excludeConvertible;

        private String m_scannerSettingPairs;

        private String m_stockTypeFilter;

        // Focal method
        public void moodyRatingBelow(String r) {
            m_moodyRatingBelow = r;
        }
    }

    // Focal class
    public class ScannerSubscriptionTest {

        // Focal class
        public ScannerSubscriptionTest() {
            MockitoAnnotations.initMocks(this);
        }

        @InjectMocks
        private ScannerSubscription scannerSubscription;

        @Mock
        private ScannerSubscription scannerSubscriptionMock;

        @Test
        public void testMoodyRatingBelow() {
            // Arrange
            String rating = "A";
            when(scannerSubscriptionMock.m_moodyRatingBelow).thenReturn(rating);
            // Act
            scannerSubscription.moodyRatingBelow(rating);
            // Assert
            verify(scannerSubscriptionMock).moodyRatingBelow(rating);
        }
    }
}
