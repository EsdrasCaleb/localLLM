package com.ib.client;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import static com.ib.client.EClientErrors.ALREADY_CONNECTED;
import static com.ib.client.EClientErrors.NO_VALID_ID;
import static com.ib.client.EClientErrors.UPDATE_TWS;
import static com.ib.client.EReader.readInt;
import static com.ib.client.EReader.readStr;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.net.Socket;
import com.ib.client.EClientErrors.CodeMsgPair;

@ExtendWith(MockitoExtension.class)
public class EClientSocket_faMsgTypeName_0_2_Test {

    @Mock
    private EClientSocket focal;

    @Test
    public void testFaMsgTypeName_ValidInput_ReturnsCorrectMessageTypeName() {
        // Arrange
        int faDataType = 1;
        String expectedMessageTypeName = "REQ_MKT_DATA";
        // Act
        String actualMessageTypeName = focal.faMsgTypeName(faDataType);
        // Assert
        assert actualMessageTypeName.equals(expectedMessageTypeName);
    }

    @Test
    public void testFaMsgTypeName_InvalidInput_ThrowsNullPointerException() {
        // Arrange
        int faDataType = -1;
        // Act and Assert
        assertThrows(NullPointerException.class, () -> focal.faMsgTypeName(faDataType));
    }

    @Test
    public void testFaMsgTypeName_OutOfRangeInput_ReturnsEmptyString() {
        // Arrange
        int faDataType = 100;
        // Act
        String actualMessageTypeName = focal.faMsgTypeName(faDataType);
        // Assert
        assert actualMessageTypeName.isEmpty();
    }
}
