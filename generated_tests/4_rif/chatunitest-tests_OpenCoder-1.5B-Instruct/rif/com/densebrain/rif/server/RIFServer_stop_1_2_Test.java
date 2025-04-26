package com.densebrain.rif.server;

import org.mockito.junit.jupiter.MockitoSettings;
import org.mockito.quality.Strictness;
import java.rmi.RemoteException;
import java.util.concurrent.TimeUnit;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.net.InetAddress;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import com.densebrain.rif.server.transport.WebServiceContainer;

@MockitoSettings(strictness = Strictness.LENIENT)
class RIFServer_stop_1_2_Test {

    @InjectMocks
    @Spy
    RIFServer server;

    @Test
    void testStop() throws InterruptedException {
        // Arrange
        String expectedMessage = "Server stopped successfully";
        String actualMessage = "";
        // Act
        try {
            server.stop();
            actualMessage = server.toString();
        } catch (RemoteException e) {
            actualMessage = e.getMessage();
        }
        // Assert
        assertEquals(expectedMessage, actualMessage);
    }
}
