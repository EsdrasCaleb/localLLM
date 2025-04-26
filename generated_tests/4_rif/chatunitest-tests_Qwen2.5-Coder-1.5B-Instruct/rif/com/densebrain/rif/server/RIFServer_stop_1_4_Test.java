package com.densebrain.rif.server;

import java.rmi.RemoteException;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import com.densebrain.rif.server.transport.WebServiceContainer;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import java.net.InetAddress;

@ExtendWith(MockitoExtension.class)
public class RIFServer_stop_1_4_Test {

    @Mock
    private WebServiceContainer mockContainer;

    @InjectMocks
    private RIFServer server;

    @Test
    public void testStop() throws Exception {
        // Arrange
        server.stop();
        // Assert
        verify(mockContainer).stopContainer();
    }
}
