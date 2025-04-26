package com.densebrain.rif.server;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.lang.reflect.Field;
import java.rmi.RemoteException;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import java.net.InetAddress;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import com.densebrain.rif.server.transport.WebServiceContainer;

@ExtendWith(MockitoExtension.class)
public class RIFServer_start_0_0_Test {

    @Mock
    private com.densebrain.rif.server.transport.WebServiceContainer container;

    @InjectMocks
    private RIFServer rifServer;

    @Test
    void testStart() throws RemoteException, NoSuchFieldException, IllegalAccessException {
        // Arrange
        // Initialize with a port
        rifServer = new RIFServer(8080);
        Field containerField = RIFServer.class.getDeclaredField("container");
        containerField.setAccessible(true);
        containerField.set(rifServer, container);
        // Act
        rifServer.start();
        // Assert
        verify(container).startContainer();
    }

    @Test
    void testStart_NullContainer() throws NoSuchFieldException, IllegalAccessException, RemoteException {
        // Arrange
        rifServer = new RIFServer(8080);
        Field containerField = RIFServer.class.getDeclaredField("container");
        containerField.setAccessible(true);
        containerField.set(rifServer, null);
        // Act & Assert
        Exception exception = assertThrows(NullPointerException.class, () -> rifServer.start());
        assertEquals("container cannot be null", exception.getMessage());
    }
}
