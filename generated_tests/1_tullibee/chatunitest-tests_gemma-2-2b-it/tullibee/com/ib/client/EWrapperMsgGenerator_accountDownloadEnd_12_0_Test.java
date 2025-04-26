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
    void testAccountDownloadEnd() {
        EWrapperMsgGenerator eWrapperMsgGenerator = new EWrapperMsgGenerator();
        String result = eWrapperMsgGenerator.accountDownloadEnd("testAccount");
        assertEquals("accountDownloadEnd: testAccount", result);
    }
}
