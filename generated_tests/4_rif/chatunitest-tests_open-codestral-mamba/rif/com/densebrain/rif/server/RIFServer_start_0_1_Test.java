package com.densebrain.rif.server;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.net.InetAddress;
import java.rmi.RemoteException;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import com.densebrain.rif.server.transport.WebServiceContainer;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

@ExtendWith(MockitoExtension.class)
public class RIFServer_start_0_1_Test {

    @Mock
    private WebServiceContainer container;

    @InjectMocks
    private RIFServer rifServer;

    @BeforeEach
    public void setUp() {
        MockitoAnnotations.initMocks(this);
    }

    @Test
    public void testStartMethod() throws RemoteException {
        // Mocking the behavior of the container's start method
        doNothing().when(container).start();
        // Invoking the start method of the rifServer
        rifServer.start();
        // Verifying that the container's start method was called once
        verify(container, times(1)).start();
    }

    @Test
    public void testStart() throws RemoteException {
        // Arrange
        doNothing().when(container).startContainer();
        // Act
        rifServer.start();
        // Assert
        verify(container).startContainer();
    }
}
