package com.densebrain.rif.server;

import java.rmi.RemoteException;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.net.InetAddress;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import com.densebrain.rif.server.transport.WebServiceContainer;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

@ExtendWith(MockitoExtension.class)
public class RIFServer_stop_1_0_Test {

    @Mock
    private WebServiceContainer container;

    @InjectMocks
    private RIFServer rifServer;

    @BeforeEach
    public void setUp() {
        // No need to call MockitoAnnotations.openMocks(this) as @ExtendWith(MockitoExtension.class) will do it
    }

    @Test
    public void testStop() throws RemoteException {
        // Arrange
        doNothing().when(container).stop();
        // Act
        rifServer.stop();
        // Assert
        verify(container, times(1)).stop();
    }

    @Test
    public void testStopWithException() throws RemoteException {
        doThrow(new RemoteException("Error stopping container")).when(container).stopContainer();
        assertThrows(RemoteException.class, () -> rifServer.stop());
        verify(container, times(1)).stopContainer();
    }
}
