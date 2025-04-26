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

public class EWrapperMsgGenerator_accountDownloadEnd_12_0_Test {

    @Test
    public void testAccountDownloadEnd() {
        String accountName = "TestAccount";
        String expectedMessage = "accountDownloadEnd: " + accountName;
        String actualMessage = EWrapperMsgGenerator.accountDownloadEnd(accountName);
        assertEquals(expectedMessage, actualMessage);
    }
}
