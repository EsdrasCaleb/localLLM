package com.densebrain.rif.server;

import java.rmi.RemoteException;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import com.densebrain.rif.server.transport.WebServiceContainer;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import java.net.InetAddress;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

@ExtendWith(MockitoExtension.class)
public class RIFServer_start_0_1_Test {

    @Mock
    private WebServiceContainer mockContainer;

    @InjectMocks
    private RIFServer rifServer;

    @BeforeEach
    public void setUp() {
        MockitoAnnotations.openMocks(this);
    }

    @Test
    public void testStartServer_Success() throws Exception {
        // No return value for startContainer, just call it
        rifServer.start();
        verify(mockContainer).startContainer();
    }

    @Test
    public void testStartServer_Exception() throws Exception {
        doThrow(new RemoteException("Error starting")).when(mockContainer).startContainer();
        Exception exception = assertThrows(RemoteException.class, () -> {
            rifServer.start();
        });
        assertEquals("Error starting", exception.getMessage());
        verify(mockContainer).startContainer();
    }

    @Test
    public void testStopServer_Success() throws Exception {
        // No return value for stopContainer, just call it
        rifServer.stop();
        verify(mockContainer).stopContainer();
    }

    @Test
    public void testStopServer_Exception() throws Exception {
        doThrow(new RemoteException("Container stop failed")).when(mockContainer).stopContainer();
        Exception exception = assertThrows(RemoteException.class, () -> rifServer.stop());
        assertEquals("Container stop failed", exception.getMessage());
        verify(mockContainer).stopContainer();
    }
}
