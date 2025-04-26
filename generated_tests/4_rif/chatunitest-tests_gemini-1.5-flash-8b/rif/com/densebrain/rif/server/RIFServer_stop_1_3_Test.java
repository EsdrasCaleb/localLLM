package com.densebrain.rif.server;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import java.rmi.RemoteException;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import com.densebrain.rif.server.transport.WebServiceContainer;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import java.net.InetAddress;

@ExtendWith(MockitoExtension.class)
class RIFServer_stop_1_3_Test {

    @Mock
    private WebServiceContainer container;

    @InjectMocks
    private RIFServer rifServer;

    @BeforeEach
    void setup() {
        MockitoAnnotations.openMocks(this);
        // Initialize with a port
        rifServer = new RIFServer(8080);
    }

    @Test
    void testStop_Success() throws RemoteException {
        // Arrange
        doNothing().when(container).stopContainer();
        // Act
        rifServer.stop();
        // Assert
        verify(container, times(1)).stopContainer();
    }

    @Test
    void testStop_ThrowsRemoteException() throws RemoteException {
        // Arrange
        RemoteException expectedException = new RemoteException("Simulated exception");
        doThrow(expectedException).when(container).stopContainer();
        // Act & Assert
        try {
            rifServer.stop();
            fail("Expected RemoteException was not thrown.");
        } catch (RemoteException e) {
            assertEquals(expectedException.getMessage(), e.getMessage());
        }
    }
}
