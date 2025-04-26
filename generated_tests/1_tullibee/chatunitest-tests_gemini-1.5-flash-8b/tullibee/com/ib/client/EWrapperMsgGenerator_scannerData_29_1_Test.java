package com.ib.client;

import com.ib.client.Contract;
import com.ib.client.ContractDetails;
import com.ib.client.EWrapperMsgGenerator;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.time.LocalDate;
import java.util.Vector;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import java.text.DateFormat;
import java.util.Date;

@ExtendWith(MockitoExtension.class)
class EWrapperMsgGenerator_scannerData_29_1_Test {

    @Mock
    private EWrapperMsgGenerator eWrapperMsgGenerator;

    @InjectMocks
    private EWrapperMsgGenerator eWrapperMsgGeneratorImpl;

    private Contract contract;

    private ContractDetails contractDetails;

    @BeforeEach
    void setUp() {
        contract = new Contract();
        contract.m_symbol = "SPY";
        contract.m_secType = "STK";
        contract.m_expiry = LocalDate.now().toString();
        contract.m_strike = 150.0;
        contract.m_right = "CALL";
        contract.m_exchange = "SMART";
        contract.m_currency = "USD";
        contract.m_localSymbol = "SPY";
        contractDetails = new ContractDetails();
        contractDetails.m_summary = contract;
        contractDetails.m_marketName = "NASDAQ";
        contractDetails.m_tradingClass = "SPY";
    }

    @Test
    void scannerData_validInput_returnsExpectedString() {
        String distance = "10";
        String benchmark = "VIX";
        String projection = "High";
        String legsStr = "None";
        int reqId = 123;
        int rank = 1;
        String expectedOutput = "id = 123 rank=1 symbol=SPY secType=STK expiry=" + LocalDate.now() + " strike=150.0 right=CALL exchange=SMART currency=USD localSymbol=SPY marketName=NASDAQ tradingClass=SPY distance=10 benchmark=VIX projection=High legsStr=None";
        String actualOutput = EWrapperMsgGenerator.scannerData(reqId, rank, contractDetails, distance, benchmark, projection, legsStr);
        assertEquals(expectedOutput, actualOutput);
    }
}
