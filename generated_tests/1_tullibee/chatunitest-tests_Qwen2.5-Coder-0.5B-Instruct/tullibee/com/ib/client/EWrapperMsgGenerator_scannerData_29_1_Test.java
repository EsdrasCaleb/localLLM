package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.text.DateFormat;
import java.util.Date;
import java.util.Vector;

class EWrapperMsgGenerator_scannerData_29_1_Test {

    @InjectMocks
    private EWrapperMsgGenerator eWrapperMsgGenerator;

    @Mock
    private ContractDetails contractDetails;

    @BeforeEach
    public void setUp() {
        MockitoAnnotations.openMocks(this);
    }

    @Test
    public void scannerDataTest() {
        // Arrange
        int reqId = 12345;
        int rank = 67890;
        ContractDetails contractDetails = new ContractDetails();
        contractDetails.m_summary = new Contract();
        String distance = "10000";
        String benchmark = "EUR";
        String projection = "Annual";
        String legsStr = "Forward";
        // Act
        String result = eWrapperMsgGenerator.scannerData(reqId, rank, contractDetails, distance, benchmark, projection, legsStr);
        // Assert
        assertEquals("id = 12345 rank=67890 symbol=EUR secType=Forward expiry=10000 strike=10000 right=Forward exchange=EUR currency=EUR localSymbol=EUR marketName=EUR tradingClass=Forward distance=10000 benchmark=EUR projection=Annual legsStr=Forward", result);
    }
}
