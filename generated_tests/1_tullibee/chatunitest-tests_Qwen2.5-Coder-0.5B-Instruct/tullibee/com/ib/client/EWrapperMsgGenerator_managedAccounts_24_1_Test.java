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

class EWrapperMsgGenerator_managedAccounts_24_1_Test {

    private EWrapperMsgGenerator wrapper;

    @BeforeEach
    void setUp() {
        wrapper = new EWrapperMsgGenerator();
    }

    @Test
    void testManagedAccounts() {
        // Arrange
        String accountsList = "Account1, Account2, Account3";
        // Act
        String result = wrapper.managedAccounts(accountsList);
        // Assert
        assertEquals("Connected : The list of managed accounts are : [Account1, Account2, Account3]", result);
    }
}
