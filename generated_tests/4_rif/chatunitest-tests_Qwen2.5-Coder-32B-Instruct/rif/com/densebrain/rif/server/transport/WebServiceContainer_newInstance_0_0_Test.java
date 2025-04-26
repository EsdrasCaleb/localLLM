package com.densebrain.rif.server.transport;

import org.apache.axis2.AxisFault;
import org.apache.axis2.context.ConfigurationContext;
import org.apache.axis2.context.ConfigurationContextFactory;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.lang.reflect.Method;
import java.rmi.RemoteException;
import java.util.List;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import java.net.InetAddress;
import java.util.LinkedList;
import org.apache.axis2.addressing.EndpointReference;
import org.apache.axis2.description.AxisService;
import org.apache.axis2.rpc.receivers.RPCMessageReceiver;
import org.apache.axis2.transport.http.turnup.SimpleHTTPServer;

@ExtendWith(MockitoExtension.class)
public class WebServiceContainer_newInstance_0_0_Test {

    @Mock
    private ConfigurationContextFactory configurationContextFactory;

    @Mock
    private ConfigurationContext configurationContext;

    @Mock
    private Log logger;

    @InjectMocks
    private WebServiceContainer webServiceContainer;

    @BeforeEach
    public void setUp() {
        MockitoAnnotations.openMocks(this);
        System.setProperty("org.apache.commons.logging.Log", "org.apache.commons.logging.impl.NoOpLog");
    }

    @Test
    public void testNewInstance_Success() throws Exception {
        // Arrange
        String hostName = "localhost";
        int port = 8080;
        String contextPath = "/context";
        when(ConfigurationContextFactory.createConfigurationContextFromFileSystem(null, null)).thenReturn(configurationContext);
        // Act
        WebServiceContainer result = WebServiceContainer.newInstance(hostName, port, contextPath);
        // Assert
        assertNotNull(result);
        assertEquals(configurationContext, getField(result, "configurationContext"));
        assertEquals(contextPath, getField(result, "contextPath"));
        assertEquals(hostName, getField(result, "hostName"));
        assertEquals(port, getField(result, "port"));
    }

    @Test
    public void testNewInstance_AxisFault() throws Exception {
        // Arrange
        String hostName = "localhost";
        int port = 8080;
        String contextPath = "/context";
        AxisFault axisFault = new AxisFault("AxisFault occurred");
        when(ConfigurationContextFactory.createConfigurationContextFromFileSystem(null, null)).thenThrow(axisFault);
        // Act & Assert
        Exception exception = assertThrows(AxisFault.class, () -> {
            WebServiceContainer.newInstance(hostName, port, contextPath);
        });
        assertEquals(axisFault.getMessage(), exception.getMessage());
    }

    @Test
    public void testNewInstance_RemoteException() throws Exception {
        // Arrange
        String hostName = "localhost";
        int port = 8080;
        String contextPath = "/context";
        RemoteException remoteException = new RemoteException("RemoteException occurred");
        when(ConfigurationContextFactory.createConfigurationContextFromFileSystem(null, null)).thenThrow(remoteException);
        // Act & Assert
        Exception exception = assertThrows(RemoteException.class, () -> {
            WebServiceContainer.newInstance(hostName, port, contextPath);
        });
        assertEquals(remoteException.getMessage(), exception.getMessage());
    }

    private Object getField(Object obj, String fieldName) throws NoSuchFieldException, IllegalAccessException {
        java.lang.reflect.Field field = obj.getClass().getDeclaredField(fieldName);
        field.setAccessible(true);
        return field.get(obj);
    }
}
