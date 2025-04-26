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

public class EWrapperMsgGenerator_managedAccounts_24_0_Test {

    @Test
    public void testManagedAccounts_WithValidInput() {
        // Arrange
        String accountsList = "Account1, Account2, Account3";
        // Act
        String result = EWrapperMsgGenerator.managedAccounts(accountsList);
        // Assert
        assertEquals("Connected : The list of managed accounts are : [Account1, Account2, Account3]", result);
    }

    @Test
    public void testManagedAccounts_EmptyInput() {
        // Arrange
        String accountsList = "";
        // Act
        String result = EWrapperMsgGenerator.managedAccounts(accountsList);
        // Assert
        assertEquals("Connected : The list of managed accounts are : []", result);
    }

    @Test
    public void testManagedAccounts_NullInput() {
        // Arrange
        String accountsList = null;
        // Act
        String result = EWrapperMsgGenerator.managedAccounts(accountsList);
        // Assert
        assertEquals("Connected : The list of managed accounts are : [null]", result);
    }
}
